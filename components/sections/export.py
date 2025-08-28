import streamlit as st
import pandas as pd
import dask.dataframe as dd
import io
import json
import gzip
import openpyxl
import numpy as np
import sqlite3
import yaml
import joblib
import tempfile
import os
from sklearn.pipeline import Pipeline
from utils.stats_utils import compute_basic_stats, compare_stats
from utils.viz_utils import alt_histogram
from utils.data_utils import sample_for_preview
import logging
import re
import threading
from pandas.api.types import is_extension_array_dtype

# Constants
MAX_PAGE_SIZE = 1000
SAMPLE_SIZE = 1000
CHART_WIDTH = 400
CHART_HEIGHT = 300

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
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = []
    # Always initialize session_lock to prevent race conditions
    st.session_state.session_lock = threading.Lock()

def detect_pii(df):
    """Basic PII detection for common patterns (e.g., email, phone)."""
    if df.empty:
        return []  # Prevent division by zero for empty DataFrames
    pii_patterns = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b",
        "ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
        "credit_card": r"\b(?:\d{4}[ -]?){3}\d{4}\b"
    }
    pii_columns = []
    for col in df.select_dtypes(include=["object"]).columns:
        sample_size = min(1000, df.shape[0].compute() if isinstance(df, dd.DataFrame) else len(df))
        if sample_size == 0:
            continue
        sample_df = df[col].sample(frac=sample_size / df.shape[0].compute(), random_state=42).compute() if isinstance(df, dd.DataFrame) else df[col].head(sample_size)
        for pii_type, pattern in pii_patterns.items():
            if sample_df.astype(str).str.contains(pattern, regex=True, na=False).any():
                pii_columns.append((col, pii_type))
    return pii_columns

def mask_pii(df, pii_columns):
    """Mask detected PII columns with [REDACTED]."""
    df_copy = df.copy()
    for col, _ in pii_columns:
        if isinstance(df_copy, dd.DataFrame):
            df_copy[col] = df_copy[col].map(lambda x: '[REDACTED]' if pd.notna(x) else x, meta=(col, 'object'))
        else:
            df_copy[col] = df_copy[col].apply(lambda x: '[REDACTED]' if pd.notna(x) else x)
    return df_copy

def make_json_serializable(obj):
    """Convert non-serializable objects (e.g., extension dtypes) to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif is_extension_array_dtype(obj):
        return str(obj)
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif hasattr(obj, '__dict__'):
        return str(obj)  # Convert objects with __dict__ (e.g., estimators) to strings
    return obj

def validate_pipeline_steps(pipeline):
    """Validate that pipeline steps are compatible with sklearn Pipeline."""
    if not isinstance(pipeline, list):
        return False
    for step in pipeline:
        if not (isinstance(step, tuple) and len(step) == 2 and isinstance(step[0], str)):
            return False
        # Basic check for estimator-like objects (has fit/transform)
        if not (hasattr(step[1], 'fit') or hasattr(step[1], 'transform')):
            return False
    return True

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: pd.util.hash_pandas_object(df.head(1000)).sum(),
                          dd.DataFrame: lambda df: pd.util.hash_pandas_object(df.head(1000).compute()).sum()})
def cached_compute_basic_stats(df):
    """Cached computation of basic statistics, hashing only a sample."""
    try:
        return compute_basic_stats(df)
    except Exception as e:
        logger.error(f"Error in compute_basic_stats: {e}")
        return {}

def section_export():
    st.header("📤 Export Data")
    initialize_session_state()
    df = st.session_state.df
    raw = st.session_state.raw_df
    if df is None or raw is None:
        st.warning("Upload a dataset first.")
        return

    try:
        # Before/After Comparison
        st.subheader("Before vs. After Comparison")
        raw_stats = cached_compute_basic_stats(raw)
        after_stats = cached_compute_basic_stats(df)
        comp = compare_stats(raw_stats, after_stats)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", f"{comp['shape_after'][0]}", f"{comp['rows_change']} ({comp['rows_pct_change']:.2f}%)")
        with c2:
            st.metric("Columns", f"{comp['shape_after'][1]}", f"{comp['columns_change']}")
        with c3:
            st.metric("Missing Values", f"{comp['missing_total_after']}", f"{comp['missing_change']} ({comp['missing_pct_change']:.2f}%)")
        if comp.get("added_columns"):
            st.success(f"Added columns: {', '.join(comp['added_columns'])}")
        if comp.get("removed_columns"):
            st.warning(f"Removed columns: {', '.join(comp['removed_columns'])}")

        # Visual Comparison
        num_cols = after_stats.get("numeric_cols", [])
        if num_cols:
            st.subheader("Visual Comparison")
            col = st.selectbox("Select numeric column for comparison", num_cols, key="export_compare_col")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Before**")
                chart1 = alt_histogram(raw, col, f"Before: {col}")
                if chart1:
                    st.altair_chart(chart1, use_container_width=True)
            with c2:
                st.write("**After**")
                chart2 = alt_histogram(df, col, f"After: {col}")
                if chart2:
                    st.altair_chart(chart2, use_container_width=True)

        # Paginated Data Preview
        st.subheader("Final Data Preview")
        page_size = st.slider("Rows per page", 10, MAX_PAGE_SIZE, 100, key="export_page_size")
        total_rows = df.shape[0].compute() if isinstance(df, dd.DataFrame) else len(df)
        max_page = max(1, total_rows // page_size + (1 if total_rows % page_size else 0))
        page = st.number_input("Page", 1, max_page, 1, key="export_page_num")
        start = max(0, (page - 1) * page_size)
        preview_df = df.compute().iloc[start:start + page_size] if isinstance(df, dd.DataFrame) else df.iloc[start:start + page_size]
        st.dataframe(preview_df)

        # Pipeline and Metadata Export
        st.subheader("Export Pipeline and Metadata")
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.session_state.session_lock:
                try:
                    pipeline_data = json.dumps(make_json_serializable(st.session_state.pipeline), ensure_ascii=False)
                    st.download_button(
                        "📜 Pipeline JSON",
                        data=pipeline_data,
                        file_name="preprocessing_pipeline.json",
                        mime="application/json",
                        help="Download the preprocessing pipeline as a JSON file."
                    )
                except Exception as e:
                    logger.error(f"Failed to export pipeline JSON: {e}")
                    st.error(f"Failed to export pipeline JSON: {e}. Ensure pipeline data is valid.")
        with c2:
            with st.session_state.session_lock:
                try:
                    pipeline_data_yaml = yaml.dump(make_json_serializable(st.session_state.pipeline), allow_unicode=True)
                    st.download_button(
                        "📜 Pipeline YAML",
                        data=pipeline_data_yaml,
                        file_name="preprocessing_pipeline.yaml",
                        mime="text/yaml",
                        help="Download the preprocessing pipeline as a YAML file."
                    )
                except ImportError:
                    st.error("YAML export requires pyyaml. Install it to enable this feature.")
                except Exception as e:
                    logger.error(f"Failed to export YAML pipeline: {e}")
                    st.error(f"Failed to export YAML pipeline: {e}. Ensure pipeline data is valid.")

        # Export Options
        st.header("Export Options")
        if df.empty:
            st.error("Cannot export an empty dataset.")
            return

        if total_rows > 100_000:
            st.warning("Exporting large datasets may take time and consume significant memory.")

        pii_columns = detect_pii(df)
        if pii_columns:
            st.warning(
                f"Potential PII detected in columns: {', '.join([col for col, pii_type in pii_columns])}. "
                "Consider masking sensitive data before export."
            )
            mask_pii_option = st.checkbox("Mask PII before export (replaces sensitive data with [REDACTED])", key="mask_pii")
        else:
            mask_pii_option = False
        st.warning("Exported files may contain sensitive data, including any PII. Store them securely.")

        # Column/Row Subset Selection
        st.subheader("Customize Export")
        column_option = st.radio(
            "Column selection",
            ["All columns", "Choose columns"],
            key="export_column_option",
            help="Select 'All columns' to export the entire dataset, or 'Choose columns' to pick specific columns."
        )
        if column_option == "Choose columns":
            export_cols = st.multiselect(
                "Select columns to export",
                df.columns.tolist(),
                default=df.columns.tolist(),
                key="export_cols",
                help="Choose specific columns to include in the export."
            )
        else:
            export_cols = df.columns.tolist()

        if not export_cols:
            st.warning("No columns selected. Please choose at least one column or select 'All columns'.")
            return
        export_df = df[export_cols]
        if mask_pii_option and pii_columns:
            export_df = mask_pii(export_df, pii_columns)

        row_filter = st.text_input(
            "Enter row filter (pandas query)",
            placeholder="e.g., column_name > 0",
            key="export_row_filter",
            help="Filter rows using a pandas query expression (e.g., 'age > 18')."
        )
        if row_filter:
            try:
                if isinstance(export_df, dd.DataFrame):
                    try:
                        export_df = export_df.query(row_filter)
                    except NotImplementedError:
                        st.warning("Complex query not supported by Dask. Converting to Pandas for filtering.")
                        export_df = export_df.compute().query(row_filter)
                else:
                    export_df = export_df.query(row_filter)
            except Exception as e:
                st.error(f"Invalid row filter: {e}")
                return

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        with c1:
            buf = io.StringIO()
            export_df.compute().to_csv(buf, index=False) if isinstance(export_df, dd.DataFrame) else export_df.to_csv(buf, index=False)
            st.download_button(
                "💾 CSV",
                data=buf.getvalue(),
                file_name="preprocessed_data.csv",
                mime="text/csv",
                help="Download the processed dataset as a CSV file."
            )
        with c2:
            try:
                import pyarrow
                if isinstance(export_df, dd.DataFrame):
                    with st.spinner("Writing Parquet file..."):
                        export_df.to_parquet("temp_parquet", index=False, engine='pyarrow')
                        with open("temp_parquet/part.0.parquet", "rb") as f:
                            buf = io.BytesIO(f.read())
                        import shutil
                        shutil.rmtree("temp_parquet")
                else:
                    buf = io.BytesIO()
                    export_df.to_parquet(buf, index=False, engine='pyarrow')
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
        with c3:
            try:
                import pyarrow
                if isinstance(export_df, dd.DataFrame):
                    with st.spinner("Writing Parquet (Snappy) file..."):
                        export_df.to_parquet("temp_parquet_snappy", index=False, engine='pyarrow', compression='snappy')
                        with open("temp_parquet_snappy/part.0.parquet", "rb") as f:
                            buf = io.BytesIO(f.read())
                        import shutil
                        shutil.rmtree("temp_parquet_snappy")
                else:
                    buf = io.BytesIO()
                    export_df.to_parquet(buf, index=False, engine='pyarrow', compression='snappy')
                st.download_button(
                    "💾 Parquet (Snappy)",
                    data=buf.getvalue(),
                    file_name="preprocessed_data_snappy.parquet",
                    mime="application/octet-stream",
                    help="Download the processed dataset as a Parquet file with Snappy compression."
                )
            except ImportError:
                st.error("Parquet export requires pyarrow. Install it to enable this feature.")
            except Exception as e:
                logger.error(f"Failed to export Parquet (Snappy): {e}")
                st.error(f"Failed to export Parquet (Snappy): {e}")
        with c4:
            buf = io.BytesIO()
            try:
                export_df.compute().to_excel(buf, index=False, engine='openpyxl') if isinstance(export_df, dd.DataFrame) else export_df.to_excel(buf, index=False, engine='openpyxl')
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
        with c5:
            buf = io.BytesIO()
            try:
                export_df.compute().to_feather(buf, compression='zstd') if isinstance(export_df, dd.DataFrame) else export_df.to_feather(buf, compression='zstd')
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
        with c6:
            buf = io.BytesIO()
            try:
                with gzip.GzipFile(fileobj=buf, mode='wb') as f:
                    export_df.compute().to_csv(f, index=False) if isinstance(export_df, dd.DataFrame) else export_df.to_csv(f, index=False)
                st.download_button(
                    "📂 CSV (Compressed)",
                    data=buf.getvalue(),
                    file_name="preprocessed_data.csv.gz",
                    mime="application/gzip",
                    help="Download the processed dataset as a compressed CSV file."
                )
            except Exception as e:
                logger.error(f"Failed to export compressed CSV: {e}")
                st.error(f"Failed to export compressed CSV: {e}")
        with c7:
            buf = io.BytesIO()
            try:
                conn = sqlite3.connect(':memory:')
                export_df.compute().to_sql('data', conn, index=False, if_exists='replace') if isinstance(export_df, dd.DataFrame) else export_df.to_sql('data', conn, index=False, if_exists='replace')
                conn.commit()
                # Create a temporary file-based SQLite database for binary export
                with tempfile.NamedTemporaryFile(delete=False, suffix='.sqlite') as temp_file:
                    backup_conn = sqlite3.connect(temp_file.name)
                    with conn:
                        conn.backup(backup_conn)
                    backup_conn.commit()
                    backup_conn.close()
                    # Read the binary contents of the temporary file
                    with open(temp_file.name, 'rb') as f:
                        buf.write(f.read())
                conn.close()
                # Clean up the temporary file
                os.unlink(temp_file.name)
                buf.seek(0)
                st.download_button(
                    "🗄️ SQLite DB",
                    data=buf.getvalue(),
                    file_name="preprocessed_data.sqlite",
                    mime="application/octet-stream",
                    help="Download the processed dataset as a binary SQLite database file."
                )
            except Exception as e:
                logger.error(f"Failed to export SQLite: {e}")
                st.error(f"Failed to export SQLite: {e}")
        
        st.caption("Export your processed dataset & pipeline. Use the Dashboard section for detailed data exploration.")
    except Exception as e:
        logger.error(f"Error in export section: {e}")
        st.error(f"Error in export section: {e}")


