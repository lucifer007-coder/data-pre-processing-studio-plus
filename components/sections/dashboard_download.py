import streamlit as st
import pandas as pd
import altair as alt
import io
import numpy as np
import openpyxl
from scipy import stats
from utils.stats_utils import compute_basic_stats
from utils.viz_utils import alt_histogram
from components.sections.recommendations import PreprocessingRecommendations
import logging
import re
import hashlib

# Constants
BIAS_THRESHOLD = 0.8
MAX_PAGE_SIZE = 1000
SAMPLE_SIZE = 1000
CHART_WIDTH = 400
CHART_HEIGHT = 300
MIN_SHAPIRO_SAMPLES = 50

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state with defaults if not set."""
    if "df" not in st.session_state:
        st.session_state.df = None
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "changelog" not in st.session_state:
        st.session_state.changelog = []

def detect_pii(df):
    """Basic PII detection for common patterns (e.g., email, phone)."""
    pii_patterns = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b",
    }
    pii_columns = []
    for col in df.select_dtypes(include=["object"]).columns:
        for pii_type, pattern in pii_patterns.items():
            if df[col].astype(str).str.contains(pattern, regex=True, na=False).any():
                pii_columns.append((col, pii_type))
    return pii_columns

def compare_stats(before, after) -> dict:
    """Lightweight version of utils.stats_utils.compare_stats for local use."""
    try:
        if not isinstance(before, dict):
            before = {}
        if not isinstance(after, dict):
            after = {}

        shape_before = before.get("shape", (0, 0))
        shape_after = after.get("shape", (0, 0))
        missing_before = before.get("missing_total", 0)
        missing_after = after.get("missing_total", 0)
        cols_before = before.get("columns", [])
        cols_after = after.get("columns", [])

        rows_change = shape_after[0] - shape_before[0]
        cols_change = shape_after[1] - shape_before[1]
        missing_change = missing_after - missing_before

        rows_pct_change = (rows_change / max(shape_before[0], 1)) * 100
        missing_pct_change = (missing_change / max(missing_before, 1)) * 100

        added_columns = list(set(cols_after) - set(cols_before))
        removed_columns = list(set(cols_before) - set(cols_after))

        return {
            "shape_before": shape_before,
            "shape_after": shape_after,
            "rows_change": rows_change,
            "rows_pct_change": round(rows_pct_change, 2),
            "columns_change": cols_change,
            "missing_total_before": int(missing_before),
            "missing_total_after": int(missing_after),
            "missing_change": int(missing_change),
            "missing_pct_change": round(missing_pct_change, 2),
            "added_columns": added_columns,
            "removed_columns": removed_columns,
        }
    except Exception as e:
        logger.error(f"Error in compare_stats: {e}")
        return {}

def compute_column_impact(raw_stats, after_stats):
    """Compute column impact metrics for mean shift and null rate change."""
    impacts = []
    before_num = raw_stats.get("describe_numeric", {})
    after_num = after_stats.get("describe_numeric", {})
    
    for col in after_num:
        if col in before_num:
            try:
                b = before_num[col]
                a = after_num[col]
                mean_shift = abs(a.get("mean", 0) - b.get("mean", 0))
                raw_null_rate = raw_stats["missing_by_col"].get(col, 0) / max(raw_stats["shape"][0], 1)
                clean_null_rate = after_stats["missing_by_col"].get(col, 0) / max(after_stats["shape"][0], 1)
                null_shift = raw_null_rate - clean_null_rate
                impacts.append({
                    "column": col,
                    "mean_shift": round(mean_shift, 4),
                    "null_rate_change": round(null_shift, 4),
                })
            except Exception as e:
                logger.error(f"Error computing impact for column {col}: {e}")
    return impacts

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: pd.util.hash_pandas_object(df).sum()})
def cached_compute_basic_stats(df):
    """Cached computation of basic statistics."""
    try:
        return compute_basic_stats(df)
    except Exception as e:
        logger.error(f"Error in compute_basic_stats: {e}")
        return {}

def section_dashboard_download():
    st.header("📊 Dashboard & Download")
    initialize_session_state()
    df = st.session_state.df
    raw = st.session_state.raw_df
    if df is None or raw is None:
        st.warning("Upload a dataset first.")
        return

    try:
        # ── 0. Compute once, reuse everywhere -------------------------
        start_time = pd.Timestamp.now()
        raw_stats = cached_compute_basic_stats(raw)
        after_stats = cached_compute_basic_stats(df)
        comp = compare_stats(raw_stats, after_stats)
        logger.info(f"Stats computation took {(pd.Timestamp.now() - start_time).total_seconds():.2f} seconds")


        # ── Before / After Comparison -----------------------------
        st.subheader("Before/After Transformations")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Before**")
            st.dataframe(raw.head())
        with col2:
            st.write("**After**")
            st.dataframe(df.head())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rows", f"{after_stats.get('shape', (0, 0))[0]}", f"{comp.get('rows_change', 0)}")
        with c2:
            st.metric("Columns", f"{after_stats.get('shape', (0, 0))[1]}", f"{comp.get('columns_change', 0)}")
        with c3:
            st.metric("Missing Values", f"{comp.get('missing_total_after', 0)}", f"{comp.get('missing_change', 0)}")
        with c4:
            st.metric("Memory (MB)", f"{after_stats.get('memory_usage_mb', 0):.2f}")

        # ──  Correlation Analysis ----------------------------------
        st.subheader("Correlation Analysis")
        num_cols = after_stats.get("numeric_cols", [])
        if num_cols:
            try:
                corr_matrix = df[num_cols].corr()
                corr_data = (
                    corr_matrix.stack()
                    .reset_index()
                    .rename(columns={0: "Correlation", "level_0": "Variable1", "level_1": "Variable2"})
                )
                chart = alt.Chart(corr_data).mark_rect().encode(
                    x=alt.X("Variable1:N", title=""),
                    y=alt.Y("Variable2:N", title=""),
                    color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
                    tooltip=["Variable1", "Variable2", "Correlation"],
                ).properties(title="Correlation Heatmap", width=CHART_WIDTH, height=CHART_HEIGHT)
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                logger.error(f"Error in correlation analysis: {e}")
        else:
            st.info("No numeric columns for correlation analysis.")

        # ── Statistical Test Results ------------------------------
        st.subheader("Statistical Test Results")
        if num_cols:
            for col in num_cols[:3]:
                try:
                    clean_series = df[col].dropna()
                    if len(clean_series) < 3:
                        st.info(f"Skipping Shapiro-Wilk test for {col}: insufficient non-NaN values.")
                        continue
                    if len(clean_series) < MIN_SHAPIRO_SAMPLES:
                        st.warning(f"Shapiro-Wilk test for {col} may be unreliable due to small sample size ({len(clean_series)} < {MIN_SHAPIRO_SAMPLES}).")
                    stat, p_value = stats.shapiro(clean_series)
                    st.write(f"**{col} Normality Test (Shapiro-Wilk)**: Statistic={stat:.3f}, p-value={p_value:.3f}")
                    if p_value < 0.05:
                        st.warning(f"{col} is not normally distributed (p < 0.05).")
                    else:
                        st.info(f"{col} appears normally distributed (p ≥ 0.05).")
                except Exception as e:
                    logger.error(f"Error in Shapiro-Wilk test for {col}: {e}")
        else:
            st.info("No numeric columns for statistical tests.")

        # ── Dashboard Tabs ---------------------------------------
        t1, t2, t3 = st.tabs(["Summary", "Distributions", "Change Log"])
        with t1:
            if comp.get("added_columns"):
                st.success(f"Added columns: {', '.join(comp['added_columns'])}")
            if comp.get("removed_columns"):
                st.warning(f"Removed columns: {', '.join(comp['removed_columns'])}")

            st.subheader("Missing by Column (After)")
            miss_after = pd.Series(after_stats.get("missing_by_col", {}))
            if miss_after.sum() > 0:
                st.dataframe(miss_after[miss_after > 0].rename("missing_count"))
            else:
                st.info("No missing values remaining!")

            st.subheader("📈 Column Impact Tracker")
            st.markdown(
                """
                **mean_shift**: Absolute change in the column’s average value after preprocessing.  
                **null_rate_change**: How much the missing-value ratio improved (positive = fewer NaNs).
                """
            )
            impacts = compute_column_impact(raw_stats, after_stats)
            if impacts:
                impacts_df = pd.DataFrame(impacts).sort_values(
                    by="mean_shift", ascending=False
                ).reset_index(drop=True)
                st.dataframe(impacts_df, use_container_width=True)
            else:
                st.info("No numeric columns to compare.")

            with st.expander("Dtypes (After)"):
                st.json(after_stats.get("dtypes", {}))
            with st.expander("Numeric Describe (After)"):
                if after_stats.get("numeric_cols", []):
                    st.dataframe(pd.DataFrame(after_stats.get("describe_numeric", {})))
                else:
                    st.info("No numeric columns present.")

            st.subheader("Full Data Preview (paginated)")
            page_size = st.slider("Rows per page", 10, MAX_PAGE_SIZE, 100, key="dash_page_size")
            total_rows = len(df)
            max_page = max(1, total_rows // page_size + (1 if total_rows % page_size else 0))
            page = st.number_input("Page", 1, max_page, 1, key="dash_page_num")
            start = max(0, (page - 1) * page_size)
            st.dataframe(df.iloc[start : start + page_size])

        with t2:
            if not num_cols:
                st.info("No numeric columns to visualize.")
            else:
                col = st.selectbox("Select numeric column", num_cols)
                a, b = st.columns(2)
                with a:
                    st.subheader("Before")
                    try:
                        chart1 = alt_histogram(raw, col, f"Before: {col}")
                        if chart1:
                            st.altair_chart(chart1, use_container_width=True)
                    except Exception as e:
                        logger.error(f"Error in histogram for {col} (before): {e}")
                with b:
                    st.subheader("After")
                    try:
                        chart2 = alt_histogram(df, col, f"After: {col}")
                        if chart2:
                            st.altair_chart(chart2, use_container_width=True)
                    except Exception as e:
                        logger.error(f"Error in histogram for {col} (after): {e}")

        with t3:
            if not st.session_state.changelog:
                st.info("No changes yet.")
            else:
                for i, msg in enumerate(st.session_state.changelog, start=1):
                    st.write(f"{i}. {msg}")

        # ── Download section ---------------------------------------
        st.markdown("---")
        st.subheader("Download Processed Data")
        pii_columns = detect_pii(df)
        if pii_columns:
            st.warning(
                f"Potential PII detected in columns: {', '.join([col for col, pii_type in pii_columns])}. "
                "Consider masking sensitive data before export."
            )
        st.warning("Exported files contain plain-text data, including any PII from the original dataset. Store them securely.")
        
        if df.empty:
            st.error("Cannot export an empty dataset.")
        else:
            if len(df) > 100_000:
                st.warning("Exporting large datasets may take time and consume significant memory.")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                buf = io.StringIO()
                df.to_csv(buf, index=False)
                st.download_button(
                    "💾 Download CSV",
                    data=buf.getvalue(),
                    file_name="preprocessed_data.csv",
                    mime="text/csv",
                    help="Download the processed dataset as a CSV file."
                )
            with col2:
                buf = io.BytesIO()
                try:
                    import pyarrow
                    df.to_parquet(buf, index=False, engine='pyarrow')
                    st.download_button(
                        "💾 Parquet",
                        data=buf.getvalue(),
                        file_name="preprocessed_data.parquet",
                        mime="application/octet-stream",
                        help="Download the processed dataset as a Parquet file."
                    )
                except ImportError:
                    st.error("Parquet export requires pyarrow. Install it to enable this feature.")
                except Exception as e:
                    logger.error(f"Failed to export Parquet: {e}")
                    st.error(f"Failed to export Parquet: {e}")
            with col3:
                buf = io.BytesIO()
                try:
                    df.to_excel(buf, index=False, engine='openpyxl')
                    st.download_button(
                        "📊 Excel",
                        data=buf.getvalue(),
                        file_name="preprocessed_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download the processed dataset as an Excel (.xlsx) file."
                    )
                except ImportError:
                    st.error("Excel export requires openpyxl. Install it to enable this feature.")
                except Exception as e:
                    logger.error(f"Failed to export Excel: {e}")
                    st.error(f"Failed to export Excel: {e}")
            with col4:
                buf = io.BytesIO()
                try:
                    df.to_feather(buf, compression='zstd')
                    st.download_button(
                        "⚡ Feather",
                        data=buf.getvalue(),
                        file_name="preprocessed_data.feather",
                        mime="application/octet-stream",
                        help="Download the processed dataset as a Feather file."
                    )
                except ImportError:
                    st.error("Feather export requires pyarrow. Install it to enable this feature.")
                except Exception as e:
                    logger.error(f"Failed to export Feather: {e}")
                    st.error(f"Failed to export Feather: {e}")

        st.caption("All processing happens in your browser session.")
    except Exception as e:
        logger.error(f"Error in dashboard section: {e}")
        st.error(f"Error in dashboard section: {e}")
