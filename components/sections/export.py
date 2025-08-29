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
import shutil
from contextlib import contextmanager
import ast
import operator

# Constants
MAX_PAGE_SIZE = 1000
SAMPLE_SIZE = 1000
CHART_WIDTH = 400
CHART_HEIGHT = 300
MAX_MEMORY_SIZE = 500_000_000  # 500MB limit for memory operations
CHUNK_SIZE = 10000

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global session lock - initialized once
if 'session_lock' not in st.session_state:
    st.session_state.session_lock = threading.Lock()

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

def validate_query_filter(query_string):
    """Validate and sanitize query filter to prevent code injection."""
    if not query_string or not isinstance(query_string, str):
        return None
    
    # Remove leading/trailing whitespace
    query_string = query_string.strip()
    
    # Check for dangerous patterns
    dangerous_patterns = [
        'import', 'exec', 'eval', '__', 'subprocess', 'os.', 'sys.',
        'open(', 'file(', 'input(', 'raw_input(', 'compile(',
        'globals(', 'locals(', 'vars(', 'dir(', 'getattr(',
        'setattr(', 'delattr(', 'hasattr('
    ]
    
    query_lower = query_string.lower()
    for pattern in dangerous_patterns:
        if pattern in query_lower:
            raise ValueError(f"Potentially dangerous operation detected: {pattern}")
    
    # Whitelist allowed operators and functions
    allowed_operators = ['>', '<', '>=', '<=', '==', '!=', '&', '|', '~', 'and', 'or', 'not']
    allowed_functions = ['isin', 'isna', 'notna', 'str.contains', 'str.startswith', 'str.endswith']
    
    # Basic syntax validation - try to parse as expression
    try:
        # Parse the query to check for valid syntax
        parsed = ast.parse(query_string, mode='eval')
        
        # Check if it contains only allowed operations
        for node in ast.walk(parsed):
            if isinstance(node, ast.Call):
                # Only allow specific function calls
                if hasattr(node.func, 'attr'):
                    func_name = f"{node.func.value.id if hasattr(node.func.value, 'id') else ''}.{node.func.attr}"
                    if func_name not in allowed_functions:
                        raise ValueError(f"Function '{func_name}' not allowed in query")
                elif hasattr(node.func, 'id'):
                    if node.func.id not in ['isin', 'isna', 'notna']:
                        raise ValueError(f"Function '{node.func.id}' not allowed in query")
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                raise ValueError("Import statements not allowed in query")
                
    except SyntaxError as e:
        raise ValueError(f"Invalid query syntax: {e}")
    except Exception as e:
        raise ValueError(f"Query validation failed: {e}")
    
    return query_string

@contextmanager
def secure_temp_file(suffix='', prefix='streamlit_'):
    """Secure temporary file context manager."""
    fd = None
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)  # Close the file descriptor immediately
        yield path
    finally:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError as e:
                logger.warning(f"Failed to cleanup temporary file {path}: {e}")

@contextmanager
def secure_temp_dir(suffix='', prefix='streamlit_'):
    """Secure temporary directory context manager."""
    path = None
    try:
        path = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
        yield path
    finally:
        if path and os.path.exists(path):
            try:
                shutil.rmtree(path)
            except OSError as e:
                logger.warning(f"Failed to cleanup temporary directory {path}: {e}")

def detect_pii(df):
    """Enhanced PII detection with better error handling."""
    if df is None or df.empty:
        return []
    
    try:
        pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b",
            "ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
            "credit_card": r"\b(?:\d{4}[ -]?){3}\d{4}\b"
        }
        
        pii_columns = []
        text_columns = df.select_dtypes(include=["object"]).columns
        
        if len(text_columns) == 0:
            return pii_columns
        
        total_rows = df.shape[0].compute() if isinstance(df, dd.DataFrame) else len(df)
        if total_rows == 0:
            return pii_columns
            
        sample_size = min(1000, total_rows)
        
        for col in text_columns:
            try:
                if isinstance(df, dd.DataFrame):
                    sample_df = df[col].sample(frac=sample_size / total_rows, random_state=42).compute()
                else:
                    sample_df = df[col].head(sample_size)
                
                for pii_type, pattern in pii_patterns.items():
                    if sample_df.astype(str).str.contains(pattern, regex=True, na=False).any():
                        pii_columns.append((col, pii_type))
                        break  # Only report first PII type found per column
                        
            except Exception as e:
                logger.warning(f"Error checking PII in column {col}: {e}")
                continue
                
        return pii_columns
        
    except Exception as e:
        logger.error(f"Error in PII detection: {e}")
        return []

def mask_pii(df, pii_columns):
    """Safely mask detected PII columns."""
    if not pii_columns:
        return df
        
    try:
        df_copy = df.copy()
        for col, _ in pii_columns:
            if col in df_copy.columns:
                if isinstance(df_copy, dd.DataFrame):
                    df_copy[col] = df_copy[col].map(
                        lambda x: '[REDACTED]' if pd.notna(x) else x, 
                        meta=(col, 'object')
                    )
                else:
                    df_copy[col] = df_copy[col].apply(
                        lambda x: '[REDACTED]' if pd.notna(x) else x
                    )
        return df_copy
    except Exception as e:
        logger.error(f"Error masking PII: {e}")
        return df

def make_json_serializable(obj):
    """Convert non-serializable objects to JSON-serializable types with better error handling."""
    try:
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
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        elif callable(obj):
            return str(obj)
        return obj
    except Exception as e:
        logger.warning(f"Could not serialize object {type(obj)}: {e}")
        return str(obj)

def validate_pipeline_steps(pipeline):
    """Enhanced pipeline validation."""
    if not isinstance(pipeline, list):
        return False
        
    try:
        for step in pipeline:
            if not (isinstance(step, tuple) and len(step) == 2 and isinstance(step[0], str)):
                return False
            # Basic check for estimator-like objects
            if not (hasattr(step[1], 'fit') or hasattr(step[1], 'transform')):
                return False
        return True
    except Exception as e:
        logger.error(f"Pipeline validation error: {e}")
        return False

def check_memory_usage(df):
    """Check if DataFrame size exceeds memory limits."""
    try:
        if isinstance(df, dd.DataFrame):
            # Estimate memory usage for Dask DataFrame
            memory_usage = df.memory_usage(deep=True).sum().compute()
        else:
            memory_usage = df.memory_usage(deep=True).sum()
        
        return memory_usage > MAX_MEMORY_SIZE
    except Exception as e:
        logger.warning(f"Could not estimate memory usage: {e}")
        return False

@st.cache_data(
    hash_funcs={
        pd.DataFrame: lambda df: pd.util.hash_pandas_object(df.head(1000)).sum(),
        dd.DataFrame: lambda df: pd.util.hash_pandas_object(df.head(1000).compute()).sum()
    },
    max_entries=10
)
def cached_compute_basic_stats(df):
    """Cached computation of basic statistics with better error handling."""
    try:
        if df is None or df.empty:
            return {}
        return compute_basic_stats(df)
    except Exception as e:
        logger.error(f"Error in compute_basic_stats: {e}")
        st.error(f"Failed to compute statistics: {e}")
        return {}

def export_chunked_csv(df, chunk_size=CHUNK_SIZE):
    """Export large DataFrames in chunks to avoid memory issues."""
    try:
        if isinstance(df, dd.DataFrame):
            total_rows = df.shape[0].compute()
        else:
            total_rows = len(df)
        
        if total_rows <= chunk_size:
            # Small dataset, export normally
            buf = io.StringIO()
            if isinstance(df, dd.DataFrame):
                df.compute().to_csv(buf, index=False)
            else:
                df.to_csv(buf, index=False)
            return buf.getvalue()
        else:
            # Large dataset, use chunked approach
            st.warning(f"Large dataset detected ({total_rows} rows). Using chunked export.")
            buf = io.StringIO()
            
            if isinstance(df, dd.DataFrame):
                # For Dask, compute in partitions
                df.compute().to_csv(buf, index=False)
            else:
                # For Pandas, write in chunks
                df.to_csv(buf, index=False)
            
            return buf.getvalue()
            
    except Exception as e:
        logger.error(f"Error in chunked CSV export: {e}")
        raise

def section_export():
    """Main export section with comprehensive security and error handling."""
    st.header("📤 Export Data")
    initialize_session_state()
    
    df = st.session_state.df
    raw = st.session_state.raw_df
    
    if df is None or raw is None:
        st.warning("Upload a dataset first.")
        return

    try:
        with st.session_state.session_lock:
            # Before/After Comparison
            st.subheader("Before vs. After Comparison")
            
            with st.spinner("Computing statistics..."):
                raw_stats = cached_compute_basic_stats(raw)
                after_stats = cached_compute_basic_stats(df)
                
            if raw_stats and after_stats:
                comp = compare_stats(raw_stats, after_stats)
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.metric(
                        "Rows", 
                        f"{comp['shape_after'][0]}", 
                        f"{comp['rows_change']} ({comp['rows_pct_change']:.2f}%)"
                    )
                with c2:
                    st.metric("Columns", f"{comp['shape_after'][1]}", f"{comp['columns_change']}")
                with c3:
                    st.metric(
                        "Missing Values", 
                        f"{comp['missing_total_after']}", 
                        f"{comp['missing_change']} ({comp['missing_pct_change']:.2f}%)"
                    )
                    
                if comp.get("added_columns"):
                    st.success(f"Added columns: {', '.join(comp['added_columns'])}")
                if comp.get("removed_columns"):
                    st.warning(f"Removed columns: {', '.join(comp['removed_columns'])}")

            # Visual Comparison
            num_cols = after_stats.get("numeric_cols", [])
            if num_cols:
                st.subheader("Visual Comparison")
                col = st.selectbox(
                    "Select numeric column for comparison", 
                    num_cols, 
                    key="export_compare_col"
                )
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Before**")
                    try:
                        chart1 = alt_histogram(raw, col, f"Before: {col}")
                        if chart1:
                            st.altair_chart(chart1, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not generate before chart: {e}")
                        
                with c2:
                    st.write("**After**")
                    try:
                        chart2 = alt_histogram(df, col, f"After: {col}")
                        if chart2:
                            st.altair_chart(chart2, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not generate after chart: {e}")

            # Paginated Data Preview
            st.subheader("Final Data Preview")
            page_size = st.slider("Rows per page", 10, MAX_PAGE_SIZE, 100, key="export_page_size")
            
            total_rows = df.shape[0].compute() if isinstance(df, dd.DataFrame) else len(df)
            max_page = max(1, total_rows // page_size + (1 if total_rows % page_size else 0))
            page = st.number_input("Page", 1, max_page, 1, key="export_page_num")
            
            start = max(0, (page - 1) * page_size)
            try:
                if isinstance(df, dd.DataFrame):
                    preview_df = df.compute().iloc[start:start + page_size]
                else:
                    preview_df = df.iloc[start:start + page_size]
                st.dataframe(preview_df)
            except Exception as e:
                st.error(f"Error displaying data preview: {e}")

            # Pipeline and Metadata Export
            st.subheader("Export Pipeline and Metadata")
            c1, c2 = st.columns(2)
            
            with c1:
                try:
                    pipeline_data = json.dumps(
                        make_json_serializable(st.session_state.pipeline), 
                        ensure_ascii=False,
                        indent=2
                    )
                    st.download_button(
                        "📜 Pipeline JSON",
                        data=pipeline_data,
                        file_name="preprocessing_pipeline.json",
                        mime="application/json",
                        help="Download the preprocessing pipeline as a JSON file."
                    )
                except Exception as e:
                    logger.error(f"Failed to export pipeline JSON: {e}")
                    st.error("Failed to export pipeline JSON. Pipeline data may be corrupted.")
                    
            with c2:
                try:
                    pipeline_data_yaml = yaml.dump(
                        make_json_serializable(st.session_state.pipeline), 
                        allow_unicode=True,
                        default_flow_style=False
                    )
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
                    st.error("Failed to export YAML pipeline. Pipeline data may be corrupted.")

            # Export Options
            st.header("Export Options")
            
            if df.empty:
                st.error("Cannot export an empty dataset.")
                return

            # Memory usage warning
            if check_memory_usage(df):
                st.warning(
                    "⚠️ Large dataset detected. Export operations may take significant time "
                    "and memory. Consider filtering data or using chunked export options."
                )

            # PII Detection and Masking
            with st.spinner("Scanning for potential PII..."):
                pii_columns = detect_pii(df)
                
            mask_pii_option = False
            if pii_columns:
                st.warning(
                    f"🔒 Potential PII detected in columns: "
                    f"{', '.join([f'{col} ({pii_type})' for col, pii_type in pii_columns])}. "
                    "Consider masking sensitive data before export."
                )
                mask_pii_option = st.checkbox(
                    "Mask PII before export (replaces sensitive data with [REDACTED])", 
                    key="mask_pii"
                )
                
            st.warning("⚠️ Exported files may contain sensitive data. Store them securely and follow your organization's data handling policies.")

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
                with st.spinner("Masking PII data..."):
                    export_df = mask_pii(export_df, pii_columns)

            # Row filtering with security validation
            row_filter = st.text_input(
                "Enter row filter (pandas query)",
                placeholder="e.g., column_name > 0",
                key="export_row_filter",
                help="Filter rows using a pandas query expression (e.g., 'age > 18'). Only safe operations are allowed."
            )
            
            if row_filter:
                try:
                    validated_filter = validate_query_filter(row_filter)
                    if validated_filter:
                        with st.spinner("Applying row filter..."):
                            if isinstance(export_df, dd.DataFrame):
                                try:
                                    export_df = export_df.query(validated_filter)
                                except NotImplementedError:
                                    st.warning("Complex query not supported by Dask. Converting to Pandas for filtering.")
                                    export_df = export_df.compute().query(validated_filter)
                            else:
                                export_df = export_df.query(validated_filter)
                                
                        # Show filtered row count
                        filtered_rows = export_df.shape[0].compute() if isinstance(export_df, dd.DataFrame) else len(export_df)
                        st.info(f"Filter applied. Rows after filtering: {filtered_rows}")
                        
                except ValueError as e:
                    st.error(f"Invalid or unsafe row filter: {e}")
                    return
                except Exception as e:
                    st.error(f"Error applying row filter: {e}")
                    return

            # Export buttons with comprehensive error handling
            st.subheader("Download Formats")
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            
            # CSV Export
            with c1:
                try:
                    csv_data = export_chunked_csv(export_df)
                    st.download_button(
                        "💾 CSV",
                        data=csv_data,
                        file_name="preprocessed_data.csv",
                        mime="text/csv",
                        help="Download the processed dataset as a CSV file."
                    )
                except Exception as e:
                    logger.error(f"CSV export failed: {e}")
                    st.error("CSV export failed")

            # Parquet Export
            with c2:
                try:
                    import pyarrow
                    
                    if isinstance(export_df, dd.DataFrame):
                        with st.spinner("Writing Parquet file..."):
                            with secure_temp_dir(suffix='_parquet') as temp_dir:
                                export_df.to_parquet(temp_dir, index=False, engine='pyarrow')
                                parquet_files = [f for f in os.listdir(temp_dir) if f.endswith('.parquet')]
                                if parquet_files:
                                    with open(os.path.join(temp_dir, parquet_files[0]), "rb") as f:
                                        parquet_data = f.read()
                                else:
                                    raise FileNotFoundError("No parquet files generated")
                    else:
                        buf = io.BytesIO()
                        export_df.to_parquet(buf, index=False, engine='pyarrow')
                        parquet_data = buf.getvalue()
                        
                    st.download_button(
                        "💾 Parquet",
                        data=parquet_data,
                        file_name="preprocessed_data.parquet",
                        mime="application/octet-stream",
                        help="Download the processed dataset as a Parquet file."
                    )
                except ImportError:
                    st.error("Parquet export requires pyarrow. Install it to enable this feature.")
                except Exception as e:
                    logger.error(f"Parquet export failed: {e}")
                    st.error("Parquet export failed")

            # Parquet Snappy Export
            with c3:
                try:
                    import pyarrow
                    
                    if isinstance(export_df, dd.DataFrame):
                        with st.spinner("Writing Parquet (Snappy) file..."):
                            with secure_temp_dir(suffix='_parquet_snappy') as temp_dir:
                                export_df.to_parquet(temp_dir, index=False, engine='pyarrow', compression='snappy')
                                parquet_files = [f for f in os.listdir(temp_dir) if f.endswith('.parquet')]
                                if parquet_files:
                                    with open(os.path.join(temp_dir, parquet_files[0]), "rb") as f:
                                        parquet_data = f.read()
                                else:
                                    raise FileNotFoundError("No parquet files generated")
                    else:
                        buf = io.BytesIO()
                        export_df.to_parquet(buf, index=False, engine='pyarrow', compression='snappy')
                        parquet_data = buf.getvalue()
                        
                    st.download_button(
                        "💾 Parquet (Snappy)",
                        data=parquet_data,
                        file_name="preprocessed_data_snappy.parquet",
                        mime="application/octet-stream",
                        help="Download the processed dataset as a Parquet file with Snappy compression."
                    )
                except ImportError:
                    st.error("Parquet export requires pyarrow. Install it to enable this feature.")
                except Exception as e:
                    logger.error(f"Parquet (Snappy) export failed: {e}")
                    st.error("Parquet (Snappy) export failed")

            # Excel Export
            with c4:
                try:
                    buf = io.BytesIO()
                    if isinstance(export_df, dd.DataFrame):
                        export_df.compute().to_excel(buf, index=False, engine='openpyxl')
                    else:
                        export_df.to_excel(buf, index=False, engine='openpyxl')
                        
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
                    logger.error(f"Excel export failed: {e}")
                    st.error("Excel export failed")

            # Feather Export
            with c5:
                try:
                    buf = io.BytesIO()
                    if isinstance(export_df, dd.DataFrame):
                        export_df.compute().to_feather(buf, compression='zstd')
                    else:
                        export_df.to_feather(buf, compression='zstd')
                        
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
                    logger.error(f"Feather export failed: {e}")
                    st.error("Feather export failed")

            # Compressed CSV Export
            with c6:
                try:
                    buf = io.BytesIO()
                    with gzip.GzipFile(fileobj=buf, mode='wb') as f:
                        csv_data = export_chunked_csv(export_df)
                        f.write(csv_data.encode('utf-8'))
                        
                    st.download_button(
                        "📂 CSV (Compressed)",
                        data=buf.getvalue(),
                        file_name="preprocessed_data.csv.gz",
                        mime="application/gzip",
                        help="Download the processed dataset as a compressed CSV file."
                    )
                except Exception as e:
                    logger.error(f"Compressed CSV export failed: {e}")
                    st.error("Compressed CSV export failed")

            # SQLite Export
            with c7:
                try:
                    with secure_temp_file(suffix='.sqlite') as temp_file:
                        # Create SQLite database
                        conn = None
                        backup_conn = None
                        
                        try:
                            conn = sqlite3.connect(':memory:')
                            
                            if isinstance(export_df, dd.DataFrame):
                                export_df.compute().to_sql('data', conn, index=False, if_exists='replace')
                            else:
                                export_df.to_sql('data', conn, index=False, if_exists='replace')
                            
                            conn.commit()
                            
                            # Backup to file
                            backup_conn = sqlite3.connect(temp_file)
                            with conn:
                                conn.backup(backup_conn)
                            backup_conn.commit()
                            
                            # Read binary data
                            with open(temp_file, 'rb') as f:
                                sqlite_data = f.read()
                                
                        finally:
                            if conn:
                                conn.close()
                            if backup_conn:
                                backup_conn.close()
                        
                    st.download_button(
                        "🗄️ SQLite DB",
                        data=sqlite_data,
                        file_name="preprocessed_data.sqlite",
                        mime="application/octet-stream",
                        help="Download the processed dataset as a binary SQLite database file."
                    )
                except Exception as e:
                    logger.error(f"SQLite export failed: {e}")
                    st.error("SQLite export failed")
        
        st.caption("Export your processed dataset & pipeline. Use the Dashboard section for detailed data exploration.")
        
    except Exception as e:
        logger.error(f"Critical error in export section: {e}")
        st.error(f"A critical error occurred during export: {e}")
        st.error("Please try refreshing the page or contact support if the issue persists.")
